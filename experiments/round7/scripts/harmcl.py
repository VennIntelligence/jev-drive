#!/usr/bin/env python3
"""HARM-CL (plans/2026-10-21-harmonizer-closedloop.md): the closed-loop paired read's extra tables beside r2.py r281. Box, repo root, op-train env.

  harmcl.py --off NTV0=F.csv R6DAG=F.csv --on NTV0=F.csv R6DAG=F.csv --base NTV0 --arm R6DAG --out results/harmonizer_closedloop.json
      on the scenes all four reads scored without a driver RPC error: ON - OFF per model (mean, zeros new / fixed), offroad zeros split by
      facts `primary` (turn inside / turn outside / junction straight / straight), junction-turn scenes (turn_deg >= 35) mean ON - OFF,
      LAUNCH-37 launched counts, and the difference in differences (arm - base)_ON - (arm - base)_OFF; paired scene bootstrap (jevdrive.stats)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "experiments/pai_fix/scripts")]
import pai_fix as PF  # noqa: E402
from jevdrive import stats  # noqa: E402

OFF = {"turn-inside": ("off_junc_inside", "off_curve_inside"), "turn-outside": ("off_junc_outside",),
       "junction-straight": ("off_junc_straight",), "straight": ("off_straight",)}


def kv(xs):
    return {k: pd.read_csv(f).set_index("scene") for k, f in (x.split("=", 1) for x in xs)}


def cnt(a, b):
    a, b = a.to_numpy(float), b.to_numpy(float); q = stats.paired(a, b); n = len(a)
    return dict(a=int(a.sum()), b=int(b.sum()), diff=n * q["mean"], lo=n * q["lo"], hi=n * q["hi"],
                new=int(((a == 1) & (b == 0)).sum()), fixed=int(((a == 0) & (b == 1)).sum()))


def main(a):
    off, on = kv(a.off), kv(a.on)
    R = {f"{k}-{v}": F for v, D in (("off", off), ("on", on)) for k, F in D.items()}
    common = sorted(set.intersection(*[set(F.index[F.score.notna() & ~F.rpc.fillna(False).astype(bool)]) for F in R.values()]))
    R = {k: F.reindex(common) for k, F in R.items()}
    n = len(common); out = dict(scenes=n, models={}, did={})
    turn = (R[f"{a.base}-off"].turn_deg.abs() >= 35).to_numpy()
    L37 = PF.launch_mask(R[f"{a.base}-off"]).to_numpy()
    for m in off:
        A, B = R[f"{m}-on"], R[f"{m}-off"]; u, v = A.score.to_numpy(float), B.score.to_numpy(float)
        d = dict(mean_on=float(u.mean()), mean_off=float(v.mean()), diff=stats.paired(u, v),
                 zeros=cnt(A.score == 0, B.score == 0), turn_scenes=int(turn.sum()), turn_diff=stats.paired(u[turn], v[turn]))
        for c, ps in OFF.items():
            d[f"off:{c}"] = cnt((A.fclass == "offroad") & A.primary.isin(ps), (B.fclass == "offroad") & B.primary.isin(ps))
        for c in PF.ZCLS + PF.PCLS:
            d[c] = cnt(A.fclass == c, B.fclass == c)
        d["launch"] = dict(scenes=int(L37.sum()), on=int((PF.launched(A).to_numpy() & L37).sum()), off=int((PF.launched(B).to_numpy() & L37).sum()))
        out["models"][m] = d
    g = lambda m, v: R[f"{m}-{v}"].score.to_numpy(float)  # noqa: E731
    don, doff = g(a.arm, "on") - g(a.base, "on"), g(a.arm, "off") - g(a.base, "off")
    out["did"] = dict(arm_minus_base_on=stats.paired(g(a.arm, "on"), g(a.base, "on")),
                      arm_minus_base_off=stats.paired(g(a.arm, "off"), g(a.base, "off")), did=stats.paired(don, doff),
                      did_turn=stats.paired(don[turn], doff[turn]))
    Path(a.out).write_text(json.dumps(out, indent=1, default=float) + "\n")
    f = lambda r: stats.fmt(r, "+.3f")  # noqa: E731
    print(f"{n} common scenes, {int(turn.sum())} junction-turn, {int(L37.sum())} launch")
    for m, d in out["models"].items():
        print(f"{m}: OFF {d['mean_off']:.3f} ON {d['mean_on']:.3f} ON-OFF {f(d['diff'])}; zeros {d['zeros']['a']} vs {d['zeros']['b']} "
              f"(new {d['zeros']['new']}, fixed {d['zeros']['fixed']}); turn {f(d['turn_diff'])}; launch {d['launch']['on']} / {d['launch']['off']}")
        for c in [*OFF, *PF.ZCLS, *PF.PCLS]:
            x = d[f"off:{c}" if c in OFF else c]
            print(f"   {c}: {x['a']} vs {x['b']} {x['diff']:+.0f} [{x['lo']:+.0f}, {x['hi']:+.0f}] new {x['new']} fixed {x['fixed']}")
    print({k: f(v) for k, v in out["did"].items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--off", nargs="+", required=True), ap.add_argument("--on", nargs="+", required=True)
    ap.add_argument("--base", default="NTV0"), ap.add_argument("--arm", default="R6DAG"), ap.add_argument("--out", required=True)
    main(ap.parse_args())
