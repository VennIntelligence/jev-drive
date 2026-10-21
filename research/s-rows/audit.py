#!/usr/bin/env python3
"""S-row audit (research/s-rows-and-paper.md; run: ssh autodl 'cd ~/data/jev-drive && bash -lc ".venv/bin/python -"' < research/s-rows/audit.py): composition, draws and hinge activity of the S pool of the current PAI lead R6DAG.
Box, repo root, CPU only. Inputs are stored files; nothing is trained or rolled.
  - draws: R6DAG.npz (names, w, grp) over on1_JTPHD; batch = 51 navtrain + 13 yaw-rate + 72 pool rows, 9 500 steps
  - per-row terms of NTV0-L18-s0 (R6DAG's init) on every S row: practice_nuplan lat_diag trainee_NTV0-L18-s0.parquet (on1_JTX S rows)
  - rows.parquet (same rows): stored rolling-policy plan hinges, nearest object, human 4 s heading change
  - onpol_nuplan rows meta (e0 = lateral offset to the log, run, scene, seed, k), source-scene outcome: first, on the box,
    experiments/alpasim_long/scripts/long_read.py collect --runs <tag>=<runs/onpol_long/roll/<tag>/c*/run,..> for RB0-F-s0 RTB1-F-s0 YS-L18-s0
    YS-L18-s1 YR10m10-F-s0 --out $DATA_DIR/runs/s_rows_audit/outcome (AlpaSim venv, < 1 min)
  -> stdout tables, $DATA_DIR/runs/s_rows_audit/{audit.json, rows_audit.parquet}
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

D = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs")) / "runs"
LD = D / "practice_nuplan/lat_diag"
OUT = D / "s_rows_audit"
PARTS = dict(rb0="L18RB0", rtb1="L18RTB1", ys="L18YS", y="L18Y")
SRC = dict(rb0="RB0-F-s0", rtb1="RTB1-F-s0", ys="YS-L18-s0/s1", y="YR10m10-F-s0")
TERMS = ["agent", "road", "lat_side", "lat_onc", "lat_soft"]
K_POOL, STEPS = 72, 9500

w = np.load(D / "round6/dag/w/R6DAG.npz")
tab = np.load(D / "op_parity/cache/on1_JTPHD/tab.npz")
assert (w["names"] == tab["names"]).all()
W = pd.DataFrame(dict(name=w["names"], w=w["w"], wgrp=w["grp"], pool=tab["pool"], mode=tab["mode"], speed=tab["speed"]))
comp = W.groupby("pool").agg(rows=("w", "size"), share=("w", "sum"), live=("w", lambda x: int((x > 0).sum())))
comp["per_batch"] = comp.share * K_POOL
comp["draws_per_live_row"] = comp.share * K_POOL * STEPS / comp.live.clip(lower=1)

t = pd.read_parquet(LD / "trainee_NTV0-L18-s0.parquet")
r = pd.read_parquet(LD / "rows.parquet")
assert (t.name.values == r.name.values).all()
t = t.join(r[["grp", "k", "obj_d", "obj_corridor", "obj_move4", "fut_dpsi4", "plan_agent", "plan_road", "fut_arc4", "off_t0", "fut_agent", "fut_road",
             "fut_ok"]])
t = t[t.pool.isin(PARTS)].copy()
try:
    rb = pd.read_parquet(LD / "trainee_RB0-F-s0.parquet").set_index("name")
    for c in ("agent", "road"):
        t[f"rb0_{c}"] = rb.loc[t.name, c].values
except Exception as e:  # noqa: BLE001
    print("no RB0 trainee", e, file=sys.stderr)
t = t.merge(W[["name", "w"]], on="name", how="left")
assert t.w.notna().all()
t["draws"] = t.w * K_POOL * STEPS
for c in TERMS:
    t[f"a_{c}"] = t[c] > 0
t["a_any"] = t[[f"a_{c}" for c in TERMS]].any(axis=1)
t["mass"] = t[TERMS].sum(axis=1)

# meta: e0, lead, run, scene, seed
ms = []
for p, d in PARTS.items():
    z = np.load(D / "onpol_nuplan/rows" / d / "meta.npz")
    m = pd.DataFrame(dict(scene=z["scene"], tag=z["tag"], seed=z["seed"], e0=z["e0"], lead=z["lead"], s0=z["s0"]))
    m["name"] = [f"on:{p}:{n[3:]}" if n.startswith("on:") else n for n in z["names"]]
    ms.append(m)
M = pd.concat(ms)
hit = t.name.isin(M.name).mean()
if hit < 0.99:   # names in the merged pool carry a part prefix; fall back to scene:seed:k parsing
    M["key"] = M.scene + ":" + M.seed.astype(str)
t["key"] = t.name.str.split(":").str[2] + ":" + t.name.str.split(":").str[3]
if hit >= 0.99:
    t = t.merge(M[["name", "scene", "tag", "e0", "lead"]], on="name", how="left")
else:
    m2 = []
    for p, d in PARTS.items():
        z = np.load(D / "onpol_nuplan/rows" / d / "meta.npz")
        m2.append(pd.DataFrame(dict(pool=p, key=[f"{s}:{q}:{k}" for s, q, k in zip(z["scene"], z["seed"], z["k"])], scene=z["scene"], tag=z["tag"],
                                     e0=z["e0"], lead=z["lead"])))
    m2 = pd.concat(m2)
    t["key"] = t.name.str.split(":", n=2).str[2]
    t = t.merge(m2, on=["pool", "key"], how="left")
print("meta join", float(t.e0.notna().mean()), file=sys.stderr)

# source-scene outcome (long_read collect per_scene.csv), if present
oc = OUT / "outcome/per_scene.csv"
if oc.exists():
    o = pd.read_csv(oc)
    o = o.rename(columns={"label": "tag"})
    o["tag"] = o.tag.str.replace(r"@.*$", "", regex=True)
    t = t.merge(o[["tag", "scene", "score"]], on=["tag", "scene"], how="left")
    t["outcome"] = np.select([t.score.isna(), t.score <= 0, t.score < 0.999], ["n/a", "zero", "partial"], "full")
else:
    t["outcome"] = "n/a"

t["held"] = t.grp == "S_held"
t["spd"] = pd.cut(t.speed, [-1, 0.5, 3, 8, 13, 99], labels=["<0.5", "0.5-3", "3-8", "8-13", ">13"])
t["kb"] = pd.cut(t.k, [-1, 5, 11, 17], labels=["k0-5", "k6-11", "k12-17"])
lead = (t.obj_d < 20) & (t.obj_corridor > 0)
turn = t.fut_dpsi4.abs() > np.deg2rad(30)
t["sit"] = np.select([t.speed < 0.5, lead, turn], ["standing", "lead<20m", "turn>30deg"], "free/other")
t["off"] = pd.cut(t.e0.abs(), [-1, 0.5, 1.5, 99], labels=["|e0|<0.5", "0.5-1.5", ">1.5"])


def strat(g):
    d = g.draws.sum()
    res = dict(rows=len(g), row_share=len(g) / len(t), draw_share=d / t.draws.sum(),
               mass_share=g.mass.sum() / t.mass.sum(), any=g.a_any.mean(), any_draw=float((g.a_any * g.draws).sum() / max(d, 1e-9)))
    for c in TERMS:
        res[c] = g[f"a_{c}"].mean()
    res["plan_agent"] = (g.plan_agent > 0).mean()
    res["plan_road"] = (g.plan_road > 0).mean()
    return pd.Series(res)


out = {}
for col in ["pool", "held", "spd", "kb", "sit", "off", "outcome"]:
    s = t.groupby(col, observed=True).apply(strat)
    out[col] = s
    print(f"\n## by {col}\n" + s.round(4).to_string())
print("\n## all\n" + strat(t).round(4).to_string())

# concentration of hinge mass and of active rows (live rows only, draw-weighted)
lv = t[t.w > 0]
m = np.sort(lv.mass.values)[::-1]
cs = np.cumsum(m) / m.sum()
conc = {f"top{p}%": float(cs[int(len(m) * p / 100) - 1]) for p in (1, 5, 10, 20)}
conc["rows_any_live"] = float(lv.a_any.mean())
conc["draw_any_live"] = float((lv.a_any * lv.draws).sum() / lv.draws.sum())
print("\n## concentration (live rows)", json.dumps(conc, indent=1))
# per term: share of term mass from road rows off the raster at t0 (off_t0)
print("\n## off_t0 rows", float(t.off_t0.mean()), "road mass share", float(t.loc[t.off_t0, "road"].sum() / t.road.sum()))
if "rb0_agent" in t:
    print("\n## RB0-F (base, before L18) vs NTV0 activity on the same rows: agent %.4f -> %.4f, road %.4f -> %.4f" % (
        (t.rb0_agent > 0).mean(), t.a_agent.mean(), (t.rb0_road > 0).mean(), t.a_road.mean()))
    both = ((t.rb0_agent > 0) | (t.rb0_road > 0))
    print("rows active under RB0 that NTV0 satisfies:", float((both & ~(t.a_agent | t.a_road)).sum() / max(both.sum(), 1)))
print("\n## pool composition (R6DAG.npz)\n" + comp.round(4).to_string())
OUT.mkdir(parents=True, exist_ok=True)
json.dump({k: v.round(5).reset_index().astype(str).to_dict("records") for k, v in out.items()} | dict(conc=conc,
          comp=comp.round(5).reset_index().astype(str).to_dict("records")), open(OUT / "audit.json", "w"), indent=1)
t.drop(columns=[c for c in t.columns if c.startswith("a_")]).to_parquet(OUT / "rows_audit.parquet")

# learnability: rows active under RB0-F (the lineage base) that NTV0 satisfies; rows whose human-consistent path `fut` also violates
t["rb_any"] = (t.rb0_agent > 0) | (t.rb0_road > 0)
t["nt_any"] = t.a_agent | t.a_road
for col in ["outcome", "sit", "off_t0", "spd"]:
    g = t[t.rb_any].groupby(col, observed=True)
    print(f"satisfied by NTV0 among RB0-active, by {col}", (1 - g.nt_any.mean()).round(3).to_dict())
for c in ("agent", "road"):
    a, f = t[f"a_{c}"], (t[f"fut_{c}"] > 0) & t.fut_ok
    print(c, "active", round(a.mean(), 4), "fut also active", round((a & f).sum() / a.sum(), 3), "their mass share", round(t.loc[a & f, c].sum() / t[c].sum(), 3))
u = (t.nt_any & (((t.fut_agent > 0) | (t.fut_road > 0)) & t.fut_ok)) | t.off_t0
print("unlearnable proxy (off_t0, or fut also violates): rows", round(u.mean(), 4), "share of active", round(u.sum() / t.a_any.sum(), 3),
      "mass share", round(t.loc[u, "mass"].sum() / t.mass.sum(), 3), "| learnable-active share of live rows", round((t.a_any & ~u)[t.w > 0].mean(), 3))
top = t[t.mass >= t.mass.quantile(0.99)]
print("top 1 % rows: off_t0", round(top.off_t0.mean(), 3), "outcome", top.outcome.value_counts(normalize=True).round(3).to_dict(),
      "term mass", (top[TERMS].sum() / top.mass.sum()).round(3).to_dict())
print("term mass share (all rows)", (t[TERMS].sum() / t.mass.sum()).round(3).to_dict())
print(t.groupby(["pool", "outcome"]).size().unstack().to_string())
