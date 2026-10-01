"""WOD val: how much RFS can cheap post-hoc pack pieces add on top of openpilot native plans (CPU only)."""
import numpy as np, pandas as pd
from jevdrive import openloop_standing as O, waymo as W
preds, ctx = O.wod_preds()
rt, rs, sp, cl, seq = ctx["rtraj"], ctx["rscore"], ctx["speed"], ctx["cluster"], ctx["seq"]
rfs = lambda p: W.rater_feedback_score(p, rt, rs, sp)
agg = lambda v: W.rfs_by_cluster(v, cl)[0]
c, l, cv, ce = (preds[k][:, 0] for k in ("op-cinque native", "op-lebowski native", "cv", "ours cls ego K1024"))
base = rfs(c)
out = {"cinque": base, "lebowski": rfs(l), "cv": rfs(cv), "cls ego": rfs(ce), "mean(cinque,lebowski)": rfs((c + l) / 2)}
def lonscale(p, s):
    q = p.copy(); q[..., 0] *= s; return q
grid = np.round(np.arange(0.80, 1.21, 0.02), 2)
S = {s: rfs(lonscale(c, s)) for s in grid}
# 2-fold cross-fit by sequence hash: pick s on one half, apply to the other
fold = pd.util.hash_array(seq.astype(object)) % 2
cf = np.empty_like(base)
for f in (0, 1):
    tr = fold != f
    sbest = max(grid, key=lambda s: S[s][tr].mean())
    cf[fold == f] = S[sbest][fold == f]; print("fold", f, "s*", sbest)
out["cinque lon-scale (cross-fit)"] = cf
out["oracle best of {cinque,lebowski}"] = np.maximum(out["cinque"], out["lebowski"])
out["oracle best of {cinque,lebowski,cv,cls ego}"] = np.max([out[k] for k in ("cinque", "lebowski", "cv", "cls ego")], 0)
out["oracle best of cinque lon-scale grid"] = np.max(np.stack(list(S.values())), 0)
rng = np.random.default_rng(0)
for k, v in out.items():
    d = v - base; bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
    print(f"{k:45s} RFS {agg(v):.3f}  frame {v.mean():.3f}  floored {(v <= 4.0 + 1e-6).mean():.3f}  d_vs_cinque {d.mean():+.3f} [{np.percentile(bs,2.5):+.3f},{np.percentile(bs,97.5):+.3f}]")
print("grid (frame-mean):", {float(s): round(float(S[s].mean()), 3) for s in grid[::2]})
