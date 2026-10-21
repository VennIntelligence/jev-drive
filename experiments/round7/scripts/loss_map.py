#!/usr/bin/env python3
"""LOSS-MAP: where R6DAG (s0 / s1) still loses points on PAI 281 under ACC + GAP, per failure class, with what our stored reads already
solve and how the local PCS ruler (d392 / d429) weighs those scenes. Diagnosis only: stored reads, no rollouts.

  pull    (Mac)  box -> $L: pai_fix facts of every stored 281 ACC + GAP read, the d429 / d434 ruler rows (45 subjects' per-scene scores) and
                 the paimix 2PL fit (item a / b)
  table   (Mac, .venv)  $L -> experiments/round7/results/loss_map.{json,csv} + the class table printed as Markdown

Class of a lost scene (score < 0.98) = pai_fix `fclass` refined by `primary` / `stratum` (CLASSES below). "Ours, ACC + GAP" = every stored
281 read of our checkpoints under the submission serve (rows subjects ending in +AG, without +hl = holdlead, plus facts-only reads);
"any serve" adds native / ACC-only / g5 / g20 / holdlead reads. Third-party garage / go-straight never count as ours.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
L = Path(os.environ.get("LOSSDIAG", REPO / "tmp/lossdiag"))
OUT = REPO / "experiments/round7/results"
BOX = "/root/autodl-tmp/ujs"
DAG = ["R6DAG-s0", "R6DAG-s1"]
THIRD = {"garage", "go-straight"}
OK = 0.98

# (key, Chinese label, rule on a facts row); first match wins; ok rows with score < OK fall to "partial-other".
CLASSES = [
    ("overshoot", "停车点冲过（无前车停止线 / 红灯）", lambda r: r.fclass == "overshoot"),
    ("off-turn-in", "出路：转弯切内侧 / 蹭内侧路沿", lambda r: r.primary in ("off_junc_inside", "off_curve_inside")),
    ("off-turn-out", "出路：转弯外侧过宽", lambda r: r.primary == "off_junc_outside"),
    ("off-junc-straight", "出路：路口内直行段", lambda r: r.primary == "off_junc_straight"),
    ("off-straight", "出路：直路 / 车道边漂出", lambda r: r.fclass == "offroad"),
    ("drift", "漂移 / 变道出走廊", lambda r: r.fclass == "zero-other"),
    ("coll-front", "我方碰撞：撞前车 / 对向 / 静止物（AF-front）", lambda r: r.fclass == "AF-front"),
    ("coll-side", "我方碰撞：侧碰 / 骑车人", lambda r: r.fclass == "side"),
    ("route-turn", "路线该转未转", lambda r: r.fclass == "route-turn"),
    ("rear-launch", "部分分：起步慢被追尾", lambda r: r.fclass == "rear-hit" and r.stratum == "launch"),
    ("rear-move", "部分分：行驶 / 减速中被追尾", lambda r: r.fclass == "rear-hit"),
    ("slow", "部分分：偏慢 / 无故减速（未被撞）", lambda r: r.fclass == "slow"),
    ("partial-other", "部分分：其他", lambda r: True),
]
NEAR1 = ("near1", "近满分 0.98–1（进度 / 舒适度微损）")
LABEL = {k: v for k, v, _ in CLASSES} | dict([NEAR1])


def cmd_pull(a):
    (L / "facts").mkdir(parents=True, exist_ok=True)
    for src, dst in [(f"{BOX}/runs/pai_fix/facts/*-281ag.csv", L / "facts/"), (f"{BOX}/tmp/pcs_r6d/rows.csv.gz", L),
                     (f"{BOX}/tmp/pcs_r6d/out/fits/paimix.npz", L)]:
        subprocess.run(["scp", "-q", f"autodl:{src}", str(dst)], check=True)
    print("pulled to", L)


def load():
    import numpy as np
    import pandas as pd
    F = {Path(p).name.replace("-L18", "").replace("-281ag.csv", ""): pd.read_csv(p).set_index("scene")
         for p in sorted(glob.glob(str(L / "facts/*-281ag.csv")))}
    R = pd.read_csv(L / "rows.csv.gz").pivot_table(index="scene", columns="subject", values="score")
    d = np.load(L / "paimix.npz")
    IT = pd.DataFrame(dict(a=d["a"], b=d["b"]), index=d["items"])
    band = [s for s in dict.fromkeys(x.split("#")[0] for x in d["subjects"]) if s not in THIRD]
    return F, R, IT, band


def classify(r):
    if r.score >= 1:
        return "ok"
    if r.score >= OK:
        return "near1"
    return next(k for k, _, f in CLASSES if f(r))


def cmd_table(a):
    import numpy as np
    import pandas as pd
    F, R, IT, band = load()
    rpc = set().union(*(set(F[t].index[F[t].rpc]) for t in DAG))
    scenes = [s for s in F[DAG[0]].index if s not in rpc and s in F[DAG[1]].index]
    ag = [c for c in R.columns if c.endswith("+AG") and c not in ("DAG-s0+AG", "DAG-s1+AG")]
    facts_only = [t for t in F if t not in DAG and t not in ("NTV0-s0", "R2C-s0", "R2J-s0", "R3MV-s0", "R3MV-s1", "R5LC-s0", "R5LC-s1")]
    ours_ag = pd.concat([R[ag].reindex(scenes)] + [F[t].score.reindex(scenes).rename(t) for t in facts_only], axis=1)
    any_serve = pd.concat([ours_ag, R[[c for c in R.columns if c not in THIRD and c not in ag and not c.startswith("DAG-")]].reindex(scenes)], axis=1)
    third = R[list(THIRD)].reindex(scenes)
    bandR = R[[b for b in band if b in R.columns]].reindex(scenes)
    rows = []
    for s in scenes:
        x = dict(scene=s, s8=s[7:15], a=float(IT.a.get(s, np.nan)), b=float(IT.b.get(s, np.nan)),
                 ntv0=float(F["NTV0-s0"].score.get(s, np.nan)), best_ag=float(ours_ag.loc[s].max()), best_ag_read=ours_ag.loc[s].idxmax(),
                 best_any=float(any_serve.loc[s].max()), garage=float(third.loc[s, "garage"]), band_pass=float((bandR.loc[s] >= OK).mean()),
                 stratum=F[DAG[0]].stratum[s])
        for t in DAG:
            r = F[t].loc[s]
            x[f"{t}_score"], x[f"{t}_cls"], x[f"{t}_fclass"], x[f"{t}_primary"], x[f"{t}_t_ev"] = float(r.score), classify(r), r.fclass, r.primary, r.t_ev_s
        rows.append(x)
    D = pd.DataFrame(rows).set_index("scene")
    D.to_csv(OUT / "loss_map.csv", float_format="%.3f")
    n = len(D)
    out = dict(n=n, rpc_dropped=sorted(rpc), mean={t: round(D[f"{t}_score"].mean(), 4) for t in DAG},
               lost={t: round((1 - D[f"{t}_score"]).sum(), 2) for t in DAG}, ag_reads=list(ours_ag.columns), band=band, classes={})
    for k, lab in [(k, lab) for k, lab, _ in CLASSES] + [NEAR1]:
        c = {}
        both = D[(D["R6DAG-s0_cls"] == k) & (D["R6DAG-s1_cls"] == k)]
        anyc = D[(D["R6DAG-s0_cls"] == k) | (D["R6DAG-s1_cls"] == k)]
        for t in DAG:
            m = D[D[f"{t}_cls"] == k]
            sc = m[f"{t}_score"]
            oth = [u for u in DAG if u != t][0]
            best = m[["best_ag", f"{oth}_score"]].max(axis=1)      # ours under ACC + GAP, the other R6DAG seed included
            c[t] = dict(scenes=len(m), zeros=int((sc == 0).sum()), pts=round(float((1 - sc).sum()), 2),
                        mean_eq=round(float((1 - sc).sum()) / n, 4),
                        ntv0_solves=int((m.ntv0 >= OK).sum()), ag_solves=int((best >= OK).sum()),
                        other_seed_solves=int((m[f"{oth}_score"] >= OK).sum()),
                        ag_rec_pts=round(float((best - sc).clip(lower=0).sum()), 2),
                        any_solves=int((m.best_any >= OK).sum()), never=int((pd.concat([best, m.best_any], axis=1).max(axis=1) < OK).sum()),
                        imperfect_a1=int((m.a >= 1).sum()), a3=int((m.a >= 3).sum()), a1=int(((m.a >= 1) & (m.a < 3)).sum()), a0=int((m.a < 1).sum()),
                        band_pass=round(float(m.band_pass.mean()), 2) if len(m) else None,
                        garage_solves=int((m.garage >= OK).sum()))
        c["union"] = len(anyc)
        c["both"] = len(both)
        c["scenes"] = {s8: {t: round(float(D.loc[s, f"{t}_score"]), 2) for t in DAG} | dict(ntv0=round(float(D.loc[s, "ntv0"]), 2),
                       best_ag=round(float(D.loc[s, "best_ag"]), 2), best_ag_read=D.loc[s, "best_ag_read"] if D.loc[s, "best_ag"] > 0 else "-", a=round(float(D.loc[s, "a"]), 2))
                       for s, s8 in zip(anyc.index, anyc.s8)}
        out["classes"][k] = dict(label=lab, **c)
    (OUT / "loss_map.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    hdr = ("| class | s0 scenes / zeros / pts | s1 scenes / zeros / pts | both seeds / union | NTV0 solves (s0 / s1) | any ours ACC+GAP solves "
           "(s0 / s1) | recoverable pts (s0 / s1) | never solved, any serve (s0 / s1) | imperfect on a >= 1 (s0 / s1) | a >= 3 / 1-3 / < 1 (s1) | band pass | garage solves (s1) |")
    print(f"n = {n} scenes; mean s0 {out['mean']['R6DAG-s0']:.4f} / s1 {out['mean']['R6DAG-s1']:.4f}; lost pts {out['lost']}")
    print(hdr), print("|" + "---|" * (hdr.count("|") - 1))
    for k, c in out["classes"].items():
        a0, a1 = c["R6DAG-s0"], c["R6DAG-s1"]
        print(f"| {k} | {a0['scenes']} / {a0['zeros']} / {a0['pts']} | {a1['scenes']} / {a1['zeros']} / {a1['pts']} | {c['both']} / {c['union']} | "
              f"{a0['ntv0_solves']} / {a1['ntv0_solves']} | {a0['ag_solves']} / {a1['ag_solves']} | {a0['ag_rec_pts']} / {a1['ag_rec_pts']} | "
              f"{a0['never']} / {a1['never']} | {a0['imperfect_a1']} / {a1['imperfect_a1']} | {a1['a3']} / {a1['a1']} / {a1['a0']} | {a1['band_pass']} | {a1['garage_solves']} |")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)
    sp.add_parser("pull")
    sp.add_parser("table")
    a = p.parse_args()
    dict(pull=cmd_pull, table=cmd_table)[a.cmd](a)
