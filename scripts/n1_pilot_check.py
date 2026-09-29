#!/usr/bin/env python
"""N1 pilot checks P1-P3 (todos/2026-09-29-n1-scorer.md): the first N tokens of lb_n1train against the op_lb lane's data
for the same tokens (keys, GIMM frames, native plans) and against E6's hold `temporal`. Runs in envs/jevdrive.
    python scripts/n1_pilot_check.py <N>"""
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
os.environ["OPI_ROOT"] = "op_lb"
from jevdrive import navsim_heads as H, navsim_zs as Z  # noqa: E402
from jevdrive.common import data_dir  # noqa: E402
import op_interp as OPI  # noqa: E402

n = int(sys.argv[1])
rd = data_dir() / "runs" / "op_lb"
mn = json.loads((rd / "lb_n1train" / "meta.json").read_text())
ml = json.loads((rd / "lb_navtrain" / "meta.json").read_text())
z = np.load(data_dir() / "runs/skill_pack/n1/feat" / f"lb_n1train_n{n}.npz")
toks = z["tokens"].tolist()
assert toks == mn["names"][:n]
lane = {t: i for i, t in enumerate(ml["names"])}
assert all(t in lane for t in toks), "pilot tokens must all be lane tokens"
li = np.array([lane[t] for t in toks])
res = {}
# P1: keys and GIMM frames against the lane's caches
gn, gl = np.load(rd / "lb_n1train/gimm.npy", mmap_mode="r"), np.load(rd / "lb_navtrain/gimm.npy", mmap_mode="r")
kn, kl = np.load(rd / "lb_n1train/keys.npy", mmap_mode="r"), np.load(rd / "lb_navtrain/keys.npy", mmap_mode="r")
eq_g, eq_k, mad = [], [], []
for i in range(n):
    a, b = np.asarray(gn[i]), np.asarray(gl[li[i]])
    eq_g.append((a == b).mean())
    mad.append(np.abs(a.astype(int) - b).mean())
    eq_k.append((np.asarray(kn[i]) == np.asarray(kl[li[i]])).mean())
res["P1"] = {"keys_byte_equal_min": float(np.min(eq_k)), "gimm_byte_equal_mean": float(np.mean(eq_g)),
             "gimm_byte_equal_min": float(np.min(eq_g)), "gimm_mean_abs_diff": float(np.mean(mad))}
# P2: native poses against the lane's plan file through the same adapter
zp = np.load(rd / "lb_navtrain/plans/gimm@cinque.npz")
assert zp["names"].tolist() == ml["names"]
ref = np.stack([OPI.adapt(zp, li[i], ml, Z.T_OUT, **OPI.ADAPTERS["base"])[0] for i in range(n)])
e = np.linalg.norm(ref[..., :2] - z["native"][..., :2], axis=-1)
pe = np.abs(zp["plan_pos"][li] - z["plan_pos"]).max((1, 2))
res["P2"] = {"navsim_xy_max_m": float(e.max()), "navsim_xy_mean_m": float(e.mean()), "tokenmax_mean_m": float(e.max(1).mean()),
             "plan_pos_max_m": float(pe.max()), "n_over_1m": int((e.max(1) > 1).sum())}
# P3: features against E6's hold features of the same tokens
tr = H.load("navtrain", True)
pos = dict(zip(tr["tokens"].tolist(), range(len(tr["tokens"]))))
hold = tr["cinque"][[pos[t] for t in toks]]
g = z["temporal"]
c = [np.corrcoef(g[:, k], hold[:, k])[0, 1] for k in range(g.shape[1]) if g[:, k].std() > 1e-6 and hold[:, k].std() > 1e-6]
cm = np.corrcoef(g.mean(0), hold.mean(0))[0, 1]
res["P3"] = {"shape": list(g.shape), "nan_or_inf": int((~np.isfinite(g)).sum()), "corr_of_dim_means": float(cm),
             "per_dim_token_corr_median": float(np.median(c)), "norm_ratio_gimm_over_hold": float(np.linalg.norm(g, axis=1).mean()
                                                                                                / np.linalg.norm(hold, axis=1).mean()),
             "cos_token_median": float(np.median((g * hold).sum(1) / np.linalg.norm(g, axis=1) / np.linalg.norm(hold, axis=1)))}
print(json.dumps(res, indent=1))
