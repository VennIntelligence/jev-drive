"""Lateral gain (decision 88, gain_single) vs tracker-lag compensation (full, alpha 1): simulated trajectory against the
native plan on 2 000 navtest tokens (every 6th). navsim2 env -> results/tracker-precomp/gain_vs_comp.json"""
import json
import sys, pickle, numpy as np
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parent)]
import offroad_lib as L, trk_precomp as T, offroad_gain as G
T._init("lb_navtest", {"native": str(L.D / "runs/op_lb/lb_navtest/preds/gimm-cinque__base.npz")})
gf = G.gains()["gain_single"]
diag = pickle.load(open(T.OUT / "diag_lb_navtest.pkl", "rb"))
toks = sorted(T._W["cp"])[::6][:2000]
rows = []
for t in toks:
    mc = L.load_cache(T._W["cp"][t]); p = T._W["P"]["native"][t]; tg = L.dense_from_poses(p)
    g = G.apply_gain(p, gf)
    S = T.sim_ego(T._W["sim"], mc, np.stack([p, g]))
    rows.append([T.track_stats(S[0], tg), T.track_stats(S[1], tg), diag[t][("native", "full")]["comp"], float(abs(g[-1, 1] - p[-1, 1])), abs(diag[t][("native", "full")]["dev_y"][3])])
res = {}
for name, i in (("base", 0), ("gain", 1), ("comp", 2)):
    out = []
    for h, j in ((1, 0), (2, 1), (4, 3)):
        py = np.array([r[i]["plan_y"][j] for r in rows]); sy = np.array([r[i]["sim_y"][j] for r in rows]); k = np.abs(py) > 0.5
        out.append(f"slope{h}={(py[k] * sy[k]).sum() / (py[k] ** 2).sum():.3f}")
    res[name] = dict(slopes=out, pos_mean=float(np.mean([r[i]["pos_mean"] for r in rows])), lat_mean=float(np.mean([r[i]["lat_mean"] for r in rows])))
    print(name, *out, "pos_mean=%.3f" % np.mean([r[i]["pos_mean"] for r in rows]), "lat_mean=%.3f" % np.mean([r[i]["lat_mean"] for r in rows]))
print("submitted |dy| at 4 s vs plan: gain mean %.3f, comp mean %.3f" % (np.mean([r[3] for r in rows]), np.mean([r[4] for r in rows])))
res["submitted_dy4_mean"] = dict(gain=float(np.mean([r[3] for r in rows])), comp=float(np.mean([r[4] for r in rows])))
json.dump(res, open(Path(__file__).resolve().parents[1] / "results/tracker-precomp/gain_vs_comp.json", "w"), indent=1)
