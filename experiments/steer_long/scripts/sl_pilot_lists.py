"""Gains pilot scene groups (steer-long.md section 5 (1)) from the stored pai281 rollouts (tmp/steer_long/rows.parquet, rollouts.csv)."""
import zlib
import numpy as np, pandas as pd
R = pd.read_parquet("tmp/steer_long/rows.parquet", columns=["track", "label", "session", "t", "v", "wz", "A_exec"])
R = R[R.track == "pai"].sort_values(["session", "t"])
E = pd.read_csv("experiments/steer_long/results/rollouts.csv"); E = E[E.track == "pai"].set_index("session")
R["rel"] = (R.t - R.groupby("session").t.transform("min")) / 1e6 + 0.1
rows = []
for s, x in R.groupby("session"):
    rel, v, wz = x.rel.values, x.v.values, x.wz.values
    vho = v[np.abs(rel - 1.7).argmin()]
    a = x.A_exec.values[(rel > 1.7) & (rel <= 3.2)].max()
    h = np.concatenate([[0], np.cumsum(wz[1:] * np.diff(rel))])        # heading, rad
    turn = {}
    for vmax in (5, 8, 10):                                              # max |dheading| over a 4 s window with v < vmax throughout, after handoff
        best = 0.0
        for i in range(len(rel)):
            if rel[i] < 1.7: continue
            j = np.searchsorted(rel, rel[i] + 4.0)
            if j > len(rel): break
            if v[i:j].max() < vmax: best = max(best, np.abs(h[i:j] - h[i]).max())
        turn[vmax] = np.degrees(best)
    rows.append(dict(session=s, scene=E.scene[s], label=E.label[s], score=E.score[s], vho=vho, a=a, **{f"turn{k}": t for k, t in turn.items()}))
D = pd.DataFrame(rows)
D.to_csv("tmp/route_night_tokyo/pai281_handoff.csv", index=False)
sc = D.groupby("scene").agg(spin=("a", lambda a: (a > 6).any()), vho=("vho", "max"),
                            z=("score", lambda s: (s == 0).sum()), **{f"t{k}": (f"turn{k}", "max") for k in (5, 8, 10)})
zys = D[D.label.isin(["ys0", "ys1"])].groupby("scene").score.apply(lambda s: (s == 0).any())
sc["zys"] = zys
key = lambda idx: sorted(idx, key=lambda s: zlib.crc32(s.encode()))
g1 = key(sc.index[sc.spin]); print("G1 spin scenes", len(g1))
c2 = key(sc.index[~sc.spin & (sc.vho >= 20)]); print("G2 cand", len(c2))
for k in (5, 8, 10):
    c = sc[~sc.spin & (sc.vho < 20) & (sc[f"t{k}"] >= 45)]
    print(f"G3 cand v<{k}: {len(c)} (zero in ys0/1: {int(c.zys.sum())})")
# G1 all spin scenes; G2 all no-spin scenes handed off at >= 20 m/s; G3 low-speed (< 5 m/s throughout, the registered low segment) turns >= 45 deg in 4 s,
# zero (in ys0 or ys1) and clean, up to 17 each, filled from the other half when one is short.
c3 = sc[~sc.spin & (sc.vho < 20) & (sc.t5 >= 45)]
cl, zz = key(c3.index[~c3.zys]), key(c3.index[c3.zys])
ncl = min(17, len(cl)); g3 = cl[:ncl] + zz[:34 - ncl]
g2 = c2[:34]
for n, g in (("g1", g1), ("g2", g2), ("g3", g3)):
    open(f"tmp/route_night_tokyo/pilot_{n}.txt", "w").write("\n".join(g) + "\n")
B = D[D.label.isin(["ys0", "ys1"])]
for n, g in (("g1", g1), ("g2", g2), ("g3", g3)):
    b = B[B.scene.isin(g)]
    print(n, len(g), "scenes; baseline ys0+ys1: spins", int((b.a > 6).sum()), "zeros", int((b.score == 0).sum()), "mean", round(b.score.mean(), 4),
          "per label zeros", b.groupby("label").score.apply(lambda s: int((s == 0).sum())).to_dict())
